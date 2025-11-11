const { parse } = require('csv-parse/sync');
const fs = require('fs');
const path = require('path');
const csvFilePath = path.resolve(__dirname, '../data/sms_spam.csv');
console.log(`Reading from: ${csvFilePath}\n`);

try {
  const fileContent = fs.readFileSync(csvFilePath, { encoding: 'utf-8' });

  const records = parse(fileContent, {
    columns: false,
    from_line: 2,
  });
  let hamCount = 0;
  let spamCount = 0;
  let totalLength = 0;

  for (const record of records) {
    const label = record[0];
    const text = record[1];

    if (label === 'ham') {
      hamCount++;
    } else if (label === 'spam') {
      spamCount++;
    }
    if (text) {
      totalLength += text.length;
    }
  }

  const totalRecords = records.length;
  const hamPercentage = ((hamCount / totalRecords) * 100).toFixed(2);
  const spamPercentage = ((spamCount / totalRecords) * 100).toFixed(2);
  const avgLength = (totalLength / totalRecords).toFixed(2);

  console.log('EDA Report');
  console.log(`Total Messages: ${totalRecords}`);
  console.log('Class Balance:');
  console.log(`  Ham:  ${hamCount} (${hamPercentage}%)`);
  console.log(`  Spam: ${spamCount} (${spamPercentage}%)`);
  console.log(`Average Message Length: ${avgLength} characters`);
} catch (error) {
  console.error('Failed to read or parse data file:');
  console.error(error);
}
