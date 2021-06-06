import os


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if os.path.isfile('en_fixed.srt'):
    os.remove('en_fixed.srt')


with open("en.srt", encoding='utf-8') as input_file:
    lines = input_file.readlines()
    with open("en_fixed.srt", 'a', encoding='utf-8') as output_file:
        for idx, line in enumerate(list(chunks(lines, 4))):
            idx += 1
            print(line)
            # if len(line) == 4:
            _, time, description, _ = line
            output_file.write(str(idx) + '\n')
            output_file.write(time)
            output_file.write(description)
            output_file.write('\n')
            # output_file.close()
